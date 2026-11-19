// GENESIS world v2 — single-env JS port of genesis/world.py. Same constants, same PRNG draw order.
// See docs/WORLD_SPEC.md. Float64 throughout. Never use Math.random or Math.hypot here.
import C from "./world_constants.json" with { type: "json" };

export const SPEC_VERSION = C.world_spec_version;
export const N_ACTIONS = C.actions.length;
export const N_CH = 6;
export const N_MEM = 5;  // memory of Vela's own last predator percept (2.2.0)
export const N_SELF = 16;  // 3.0.0: + injury
export const OBS_DIM = C.n_rays * N_CH + N_SELF + N_MEM + N_ACTIONS;
export const CONSTANTS = C;
export const MODES = ["patrol", "hunt", "search", "rest"];
const TWO_PI = 2 * Math.PI;
const PATROL = 0, HUNT = 1, SEARCH = 2, REST = 3, A_DASH = 3, A_REST = 7;

export class Mulberry32 {
  constructor(seed) { this.s = seed >>> 0; }
  next() {
    this.s = (this.s + 0x6D2B79F5) >>> 0;
    const s = this.s;
    let t = Math.imul(s ^ (s >>> 15), s | 1);
    t = ((t + Math.imul(t ^ (t >>> 7), t | 61)) ^ t) >>> 0;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }
}

function rayAngles() {
  const R = C.n_rays, start = -C.fov / 2, stop = C.fov / 2, step = (stop - start) / (R - 1);
  const a = [];
  for (let i = 0; i < R; i++) a.push(i * step + start);
  a[R - 1] = stop;
  return a;
}
export const RAY_REL = rayAngles();
const norm2 = (x, y) => Math.sqrt(x * x + y * y);
const clamp = (x, lo, hi) => Math.min(Math.max(x, lo), hi);

export class World {
  constructor(family = "standard", consts = C) {
    this.c = consts; this.f = { ...consts.families[family] }; this.family = family;
    this.events = []; this.eventSeq = 0;
  }

  u() { return this.rng.next(); }

  samplePoint(margin, clear) {
    const W = this.c.width, H = this.c.height;
    let x = 0, y = 0;
    for (let k = 0; k < this.c.sample_attempts; k++) {
      x = margin + this.u() * (W - 2 * margin);
      y = margin + this.u() * (H - 2 * margin);
      if (clear(x, y)) return [x, y, true];
    }
    return [x, y, false];
  }
  samplePolar(cx, cy, R, clear) {
    const W = this.c.width, H = this.c.height;
    let x = 0, y = 0;
    for (let k = 0; k < this.c.sample_attempts; k++) {
      const a = this.u() * TWO_PI;
      const r = R * Math.sqrt(this.u());
      x = cx + r * Math.cos(a); y = cy + r * Math.sin(a);
      const inb = x >= 0.4 && x <= W - 0.4 && y >= 0.4 && y <= H - 0.4;
      if (inb && clear(x, y)) break;
    }
    return [clamp(x, 0.4, W - 0.4), clamp(y, 0.4, H - 0.4)];
  }

  slatesOk(x, y, extra) { for (const o of this.slates) if (o.on && !(norm2(o.x - x, o.y - y) > o.r + extra)) return false; return true; }
  sheltersOk(x, y, extra) { for (const s of this.shelters) if (s.on && !(norm2(s.x - x, s.y - y) > s.r + extra)) return false; return true; }
  poolsOk(x, y, extra) { for (const p of this.pools) if (p.on && !(norm2(p.x - x, p.y - y) > p.r + extra)) return false; return true; }
  inPool(x, y) { return !this.poolsOk(x, y, 0.0); }
  notInVents(x, y) { for (const v of this.vents) if (v.on && !(norm2(v.x - x, v.y - y) > v.r)) return false; return true; }
  ventDuration(active) { const u = this.u(); return active ? 150 + Math.floor(u * 250) : 100 + Math.floor(u * 200); }
  weatherDuration(rain) {
    const u = this.u(), c = this.c;
    return rain ? c.rain_min + Math.floor(u * c.rain_rand) : c.dry_min + Math.floor(u * c.dry_rand);
  }
  placeFood(i) {
    const rf = this.c.food_radius, m = this.meadows[i % this.c.n_meadows];
    const [x, y] = this.samplePolar(m.x, m.y, this.c.meadow_spread, (x, y) =>
      this.slatesOk(x, y, rf + 0.1) && this.notInVents(x, y) && this.sheltersOk(x, y, rf) && this.poolsOk(x, y, rf)
      && norm2(x - this.x, y - this.y) > 1.0);
    this.food[i].x = x; this.food[i].y = y;
  }
  newWaypoint() {
    const pr = this.c.predator_radius;
    const [x, y] = this.samplePoint(0.5, (x, y) => this.slatesOk(x, y, pr + 0.1) && this.sheltersOk(x, y, pr + 0.1));
    this.pred.tx = x; this.pred.ty = y;
  }
  light(t = this.t) { return 0.5 + 0.5 * Math.cos((TWO_PI * (t + this.t0)) / this.c.day_ticks); }

  reset(seed) {
    const c = this.c, f = this.f, rc = c.creature_radius, pr = c.predator_radius, gap = 2 * rc + 0.05;
    this.seed = seed; this.rng = new Mulberry32(seed);
    const mk = (n, o) => Array.from({ length: n }, () => ({ ...o }));
    this.slates = mk(c.max_obstacles, { x: -100, y: -100, r: 0, on: false });
    this.shelters = mk(c.max_shelters, { x: -100, y: -100, r: 0, on: false, h: 0, life: -1 });
    this.fissures = mk(c.max_fissures, { x: -100, y: -100, state: 0, timer: 0 });
    this.plantTimer = 0; this.injury = 0;
    this.pools = mk(c.max_pools, { x: -100, y: -100, r: 0, on: false });
    this.vents = mk(c.max_vents, { x: -100, y: -100, r: 0, on: false, present: false, timer: 0 });
    this.meadows = mk(c.n_meadows, { x: -100, y: -100 });
    this.food = mk(c.max_food, { x: -100, y: -100, on: false, timer: 0 });
    this.x = -100; this.y = -100;
    this.pred = { on: false, x: -100, y: -100, heading: 0, mode: PATROL, timer: 0, fatigue: 0, tx: 0, ty: 0, lx: 0, ly: 0, speed: 0, seen: false, pounce: false };
    let n = f.obs_min + Math.floor(this.u() * (f.obs_max - f.obs_min + 1));
    for (let j = 0; j < c.max_obstacles; j++) {
      if (j >= n) continue;
      const r = f.obs_rmin + this.u() * (f.obs_rmax - f.obs_rmin);
      const [x, y] = this.samplePoint(r + gap, (x, y) => this.slatesOk(x, y, r + gap));
      Object.assign(this.slates[j], { x, y, r, on: true });
    }
    n = f.sh_min + Math.floor(this.u() * (f.sh_max - f.sh_min + 1));
    for (let s = 0; s < c.max_shelters; s++) {
      if (s >= n) continue;
      const r = 0.9 + this.u() * 0.4, wall = r + 2 * pr + 0.05;  // 3.0.0: the predator can pass between a thicket and a wall
      const [x, y] = this.samplePoint(wall, (x, y) => this.slatesOk(x, y, r + gap) && this.sheltersOk(x, y, r + 1.0));
      Object.assign(this.shelters[s], { x, y, r, on: true });
    }
    n = f.pool_min + Math.floor(this.u() * (f.pool_max - f.pool_min + 1));
    for (let q = 0; q < c.max_pools; q++) {
      if (q >= n) continue;
      const r = 1.2 + this.u() * 0.8;
      const [x, y] = this.samplePoint(r + 0.3, (x, y) => this.slatesOk(x, y, r + gap) && this.sheltersOk(x, y, r + 0.8) && this.poolsOk(x, y, r + 1.0));
      Object.assign(this.pools[q], { x, y, r, on: true });
    }
    for (let v = 0; v < c.max_vents; v++) {
      if (v >= f.n_vents) continue;
      const r = 0.6 + this.u() * 0.3;
      const [x, y] = this.samplePoint(r, (x, y) => this.slatesOk(x, y, 0.0) && this.sheltersOk(x, y, r + 0.5) && this.poolsOk(x, y, r));
      const active = this.u() < 0.7;
      Object.assign(this.vents[v], { x, y, r, present: true, on: active });
      this.vents[v].timer = this.ventDuration(active);
    }
    const nearShelter = (x, y) => this.shelters.some((s) => s.on && norm2(s.x - x, s.y - y) - s.r <= c.meadow_shelter_max);
    const hollowOk = (x, y) => { const v = this.vents[0], dv = norm2(v.x - x, v.y - y) - v.r;  // meadow 0: the rich hollow by the first vent
      return dv >= c.hollow_vent_min && dv <= c.hollow_vent_max && this.sheltersOk(x, y, 1.0) && this.slatesOk(x, y, 0.0) && this.poolsOk(x, y, 1.0); };
    this.meadows.forEach((m, k) => {
      const [x, y] = this.samplePoint(1.5, k === 0 && f.n_vents > 0 ? hollowOk : (x, y) => this.sheltersOk(x, y, c.meadow_shelter_min) && nearShelter(x, y)
        && this.slatesOk(x, y, 0.0) && this.poolsOk(x, y, 1.0));
      m.x = x; m.y = y;
    });
    const [cx, cy] = this.samplePoint(rc + 0.1, (x, y) => {
      if (!this.slatesOk(x, y, rc + 0.1)) return false;
      for (const v of this.vents) if (v.present && !(norm2(v.x - x, v.y - y) > v.r + rc)) return false;
      return true;
    });
    this.x = cx; this.y = cy;
    this.heading = this.u() * TWO_PI;
    if (f.predator) {
      const [px, py] = this.samplePoint(0.5, (x, y) => this.slatesOk(x, y, pr + 0.1) && this.sheltersOk(x, y, pr + 0.1)
        && norm2(x - this.x, y - this.y) > 8.0);
      Object.assign(this.pred, { x: px, y: py, on: true });
      this.pred.speed = f.hunt_min + this.u() * (f.hunt_max - f.hunt_min);
      this.newWaypoint();
      Object.assign(this.pred, { mode: PATROL, timer: 0, fatigue: 0, heading: 0, lx: px, ly: py });
    }
    for (let i = 0; i < f.food_n; i++) { this.placeFood(i); this.food[i].on = true; }
    this.t0 = f.t0_fixed >= 0 ? f.t0_fixed : Math.floor(this.u() * c.day_ticks);
    this.raining = this.u() < c.rain_start_p;
    this.weatherTimer = this.weatherDuration(this.raining);
    this.inWater = this.inPool(this.x, this.y);
    if (!f.rain) { this.raining = false; this.weatherTimer = 1e9; }
    this.meadows.forEach((m) => { m.phase = Math.floor(this.u() * c.fertility_period); });
    for (const sh of this.shelters) if (sh.on) sh.h = c.thicket_h_min + this.u() * (1 - c.thicket_h_min);
    this.energy = c.energy_start; this.stamina = 1.0; this.v = 0.0; this.w = 0.0;
    this.t = 0; this.lastAction = A_REST;
    this.contact = false; this.inHazard = false; this.resting = false;
    this.inShelter = this.cover(this.x, this.y);
    this.epFood = 0; this.dead = false; this.caught = false; this.truncated = false;
    this.events = [];
    this.memT = 1000000; this.memX = 0; this.memY = 0; this.memPounce = false;
    this.updateMemory();
  }

  // ---- visitor / perturbation mutations (same clearance rules as Python)
  insertObstacle(r = null, at = null) {
    const c = this.c, f = this.f, rc = c.creature_radius, pr = c.predator_radius;
    if (this.slates.every((o) => o.on)) return false;
    if (r === null) r = f.obs_rmin + this.u() * (f.obs_rmax - f.obs_rmin);
    const clear = (x, y) => {
      if (!(this.slatesOk(x, y, r + 2 * rc + 0.05) && this.sheltersOk(x, y, r + 0.2) && this.poolsOk(x, y, r))) return false;
      if (!(norm2(x - this.x, y - this.y) > r + rc + 0.5)) return false;
      if (this.pred.on && !(norm2(x - this.pred.x, y - this.pred.y) > r + pr + 0.2)) return false;
      for (const fd of this.food) if (fd.on && !(norm2(fd.x - x, fd.y - y) > r + c.food_radius)) return false;
      return true;
    };
    let x, y, ok;
    if (at === null) [x, y, ok] = this.samplePoint(r + 2 * rc + 0.05, clear);
    else { [x, y] = at; ok = x >= r && y >= r && x <= c.width - r && y <= c.height - r && clear(x, y); }
    if (!ok) return false;
    const slot = this.slates.findIndex((o) => !o.on);
    Object.assign(this.slates[slot], { x, y, r, on: true });
    return true;
  }
  moveFood(i, x, y) {
    const c = this.c, rf = c.food_radius, m = 0.4;
    if (!this.food[i].on || x < m || y < m || x > c.width - m || y > c.height - m) return false;
    if (!(this.slatesOk(x, y, rf + 0.1) && this.notInVents(x, y) && this.sheltersOk(x, y, rf) && this.poolsOk(x, y, rf)
      && norm2(x - this.x, y - this.y) > 1.0)) return false;
    this.food[i].x = x; this.food[i].y = y; return true;
  }
  wakeVent(v) { if (!this.vents[v].present) return false; this.vents[v].on = true; this.vents[v].timer = 300; return true; }
  alertPredator(at = null) {  // a disturbance: the predator searches a point (default: where Vela is); a visitor lure picks the point
    const p = this.pred;
    if (!p.on || p.mode === REST) return false;
    p.mode = SEARCH; p.lx = at ? at[0] : this.x; p.ly = at ? at[1] : this.y; p.timer = this.c.predator_search_ticks;
    this.emit(at ? "lure" : "alert", { x: p.lx, y: p.ly }); return true;
  }
  cover(x, y) { const cm = this.c.cover_min; return this.shelters.some((s) => s.on && s.h > cm && norm2(s.x - x, s.y - y) < s.r); }
  fertility(k) {
    const c = this.c, cyc = 0.5 + 0.5 * Math.sin((TWO_PI * (this.t + this.meadows[k].phase)) / c.fertility_period);
    return c.fertility_floor + c.fertility_gain * cyc + c.rain_fertility * (this.raining ? 1 : 0);
  }
  plantSeed(at = null) {
    const c = this.c, g = c.max_food - 1, rf = c.food_radius, fd = this.food[g];
    if (fd.on || this.plantTimer > 0) return false;
    const clear = (x, y) => this.slatesOk(x, y, rf + 0.1) && this.notInVents(x, y) && this.sheltersOk(x, y, rf) && this.poolsOk(x, y, rf)
      && norm2(x - this.x, y - this.y) > 1.0;
    let x, y, ok;
    if (at === null) [x, y, ok] = this.samplePoint(0.5, clear);
    else { [x, y] = at; ok = x >= 0.5 && y >= 0.5 && x <= c.width - 0.5 && y <= c.height - 0.5 && clear(x, y); }
    if (!ok) return false;
    fd.x = x; fd.y = y; this.plantTimer = c.seed_ripen; this.emit("plant", { x, y }); return true;
  }
  raiseThicket(at = null) {
    const c = this.c, sh = this.shelters[c.max_shelters - 1], r = c.raised_radius, wall = r + 2 * c.predator_radius + 0.05;
    if (sh.on) return false;
    const clear = (x, y) => this.slatesOk(x, y, r + 0.2) && this.sheltersOk(x, y, r + 0.5) && this.poolsOk(x, y, r);
    let x, y, ok;
    if (at === null) [x, y, ok] = this.samplePoint(wall, clear);
    else { [x, y] = at; ok = x >= wall && y >= wall && x <= c.width - wall && y <= c.height - wall && clear(x, y); }
    if (!ok) return false;
    Object.assign(sh, { x, y, r, on: true, h: 0, life: c.raised_life }); this.emit("raise", { x, y }); return true;
  }
  callRain() {
    if (this.raining) return false;
    this.raining = true; this.weatherTimer = this.weatherDuration(true); this.emit("rain_start"); return true;
  }
  wakeGround(at = null) {
    const c = this.c, f = this.f, slot = this.fissures.findIndex((z) => z.state === 0);
    if (slot < 0) return false;
    let x, y;
    if (at === null) {
      if (f.n_vents <= 0) return false;
      const k = Math.min(Math.floor(this.u() * f.n_vents), f.n_vents - 1), ang = this.u() * TWO_PI, v = this.vents[k];
      const dist = v.r + c.fissure_vent_min + this.u() * (c.fissure_vent_max - c.fissure_vent_min), fr = c.fissure_radius;
      x = clamp(v.x + dist * Math.cos(ang), fr, c.width - fr); y = clamp(v.y + dist * Math.sin(ang), fr, c.height - fr);
    } else {
      [x, y] = at; const fr = c.fissure_radius;
      if (!(x >= fr && y >= fr && x <= c.width - fr && y <= c.height - fr)) return false;
    }
    Object.assign(this.fissures[slot], { x, y, state: 1, timer: c.fissure_warn }); this.emit("fissure_warning", { x, y }); return true;
  }
  perturb() {
    const f = this.f, u = this.u();
    const a1 = f.p_food, a2 = a1 + f.p_obs, a3 = a2 + f.p_vent, a4 = a3 + f.p_alert;
    const a5 = a4 + f.p_plant, a6 = a5 + f.p_raise, a7 = a6 + f.p_rain, a8 = a7 + f.p_fissure;
    if (u < a1) {
      const k = Math.min(Math.floor(this.u() * f.food_n), f.food_n - 1);
      if (this.food[k].on) this.placeFood(k);
    } else if (u < a2) this.insertObstacle();
    else if (u < a3) {
      if (f.n_vents > 0) { const k = Math.min(Math.floor(this.u() * f.n_vents), f.n_vents - 1); this.wakeVent(k); }
    } else if (u < a4) this.alertPredator();
    else if (u < a5) this.plantSeed();
    else if (u < a6) this.raiseThicket();
    else if (u < a7) this.callRain();
    else if (u < a8) this.wakeGround();
  }

  losClear(ax, ay, bx, by) {
    const abx = bx - ax, aby = by - ay, L2 = Math.max(abx * abx + aby * aby, 1e-12);
    for (const o of this.slates) {
      if (!o.on) continue;
      const acx = o.x - ax, acy = o.y - ay;
      const s = clamp((acx * abx + acy * aby) / L2, 0.0, 1.0);
      const cx = ax + s * abx - o.x, cy = ay + s * aby - o.y;
      if (!(cx * cx + cy * cy >= o.r * o.r)) return false;
    }
    return true;
  }

  static push(px, py, cx, cy, minD) {
    const dx = px - cx, dy = py - cy, d = norm2(dx, dy);
    if (!(d < minD)) return null;
    const nx = d > 1e-12 ? dx / d : 1.0, ny = d > 1e-12 ? dy / d : 0.0;
    return [cx + nx * minD, cy + ny * minD];
  }

  predatorStep() {
    const c = this.c, f = this.f, p = this.pred;
    if (!p.on) return false;
    const light = this.light();
    let detect = (c.detect_night + (c.detect_day - c.detect_night) * light) * f.detect_mult;
    if (this.raining) detect = detect * c.rain_detect;
    const vis = this.v < 0.02 ? 0.45 : this.v < 0.07 ? 0.65 : this.v < 0.15 ? 1.0 : 1.35;
    const d = norm2(this.x - p.x, this.y - p.y);
    const exposed = !this.inShelter && this.losClear(p.x, p.y, this.x, this.y);
    const seen = exposed && d < detect * vis, seenExt = exposed && d < detect * vis * c.lose_mult;
    p.seen = seen;
    let mode = p.mode;
    const toHunt = (mode === PATROL || mode === SEARCH) && seen;
    const tired = mode === HUNT && p.fatigue >= c.predator_fatigue_max;
    const lost = mode === HUNT && !tired && !seenExt;
    mode = toHunt ? HUNT : tired ? REST : lost ? SEARCH : mode;
    if (tired) p.timer = c.predator_rest_ticks; else if (lost) p.timer = c.predator_search_ticks;
    const hunt = mode === HUNT, search = mode === SEARCH, patrol = mode === PATROL, rest = mode === REST;
    if (hunt) { p.lx = this.x; p.ly = this.y; }
    const tx = hunt ? this.x : search ? p.lx : p.tx, ty = hunt ? this.y : search ? p.ly : p.ty;
    const pounce = hunt && d <= c.pounce_dist;
    p.pounce = pounce;
    let speed = pounce ? p.speed : hunt ? c.predator_stalk : search ? c.predator_search : patrol ? c.predator_patrol : 0.0;
    if (this.inPool(p.x, p.y)) speed = speed * c.predator_water_factor;
    p.fatigue = pounce ? p.fatigue + 1.0 : hunt ? p.fatigue + 0.3 : Math.max(p.fatigue - 0.5, 0.0);
    if (search || rest) p.timer -= 1;
    const dx = tx - p.x, dy = ty - p.y, dist = norm2(dx, dy), step = Math.min(speed, dist);
    if (dist > 1e-9 && step > 0) {
      p.x = p.x + (dx / dist) * step; p.y = p.y + (dy / dist) * step;
      p.heading = Math.atan2(dy, dx);
    }
    const pr = c.predator_radius;
    let px = p.x, py = p.y, q;
    for (const o of this.slates) if (o.on && (q = World.push(px, py, o.x, o.y, o.r + pr))) [px, py] = q;
    for (const s of this.shelters) if (s.on && s.h > c.cover_min && (q = World.push(px, py, s.x, s.y, s.r + pr))) [px, py] = q;
    p.x = clamp(px, pr, c.width - pr); p.y = clamp(py, pr, c.height - pr);
    const arrived = patrol && norm2(p.x - p.tx, p.y - p.ty) < 0.3;
    const endS = search && (p.timer <= 0 || norm2(p.x - p.lx, p.y - p.ly) < 0.3);
    const endR = rest && p.timer <= 0;
    if (endS || endR) mode = PATROL;
    if (endR) p.fatigue = 0.0;
    p.mode = mode;
    if (arrived || endS || endR) this.newWaypoint();
    return !this.inShelter && norm2(this.x - p.x, this.y - p.y) < c.catch_dist;
  }

  emit(kind, extra = {}) { this.events.push({ id: `${this.seed}:${this.t}:${this.eventSeq++}`, tick: this.t, kind, source: this.actSource || "nature", ...extra }); }

  step(a) {
    const c = this.c, rc = c.creature_radius;
    const canDash = this.stamina >= c.dash_min_stamina, dashing = a === A_DASH && canDash;
    let tv = a === A_DASH && !canDash ? c.action_v[0] : c.action_v[a];
    if (this.stamina < c.exhausted_stamina) tv = tv * c.exhausted_factor;
    if (this.inWater) tv = tv * c.water_factor;
    const tw = c.action_w[a];
    this.v = this.v + (tv - this.v) * c.accel_v;
    this.w = this.w + (tw - this.w) * c.accel_w;
    let h = (this.heading + this.w) % TWO_PI; if (h < 0) h += TWO_PI;
    this.heading = h;
    let px = this.x + this.v * Math.cos(h), py = this.y + this.v * Math.sin(h);
    let contact = false, q;
    for (const o of this.slates) if (o.on && (q = World.push(px, py, o.x, o.y, rc + o.r))) { if (q[0] !== px || q[1] !== py) contact = true; [px, py] = q; }
    const qx = clamp(px, rc, c.width - rc), qy = clamp(py, rc, c.height - rc);
    if (qx !== px || qy !== py) contact = true;
    this.x = qx; this.y = qy; this.contact = contact;
    const wasSheltered = this.inShelter;
    this.inShelter = this.cover(this.x, this.y);
    this.inWater = this.inPool(this.x, this.y);
    this.resting = a === A_REST && this.v < 0.02;
    this.energy -= (this.resting ? c.energy_decay_rest : c.energy_decay) + c.move_cost_k * this.v * this.v;
    const moving = a <= 2 || (a === A_DASH && !canDash), slow = a >= 4 && a <= 6;
    const dst = dashing ? c.stamina_dash : moving ? c.stamina_move : slow ? c.stamina_slow
      : a === A_REST ? c.stamina_rest + c.stamina_rest_shelter * (this.inShelter ? 1 : 0) : 0.0;
    this.stamina = clamp(this.stamina + (dst > 0 && this.injury > 0 ? dst * c.injury_recovery : dst), 0.0, 1.0);
    let reward = 0;
    let inh = false;
    for (const v of this.vents) if (v.on && norm2(v.x - this.x, v.y - this.y) < v.r) inh = true;
    for (const z of this.fissures) if (z.state === 2 && norm2(z.x - this.x, z.y - this.y) < c.fissure_radius) inh = true;
    if (inh && this.injury === 0) this.emit("injured");
    this.injury = inh ? c.injury_ticks : Math.max(this.injury - 1, 0);
    this.inHazard = inh;
    if (inh) { this.energy -= c.hazard_drain; reward += c.reward_hazard_tick; }
    for (let i = 0; i < c.max_food; i++) {
      const fd = this.food[i];
      if (fd.on && norm2(fd.x - this.x, fd.y - this.y) < rc + c.food_radius) {
        this.energy = Math.min(1.0, this.energy + c.food_gain);
        reward += c.reward_food; this.epFood += 1;
        this.emit("eat", { index: i, x: fd.x, y: fd.y });
        fd.on = false;
        fd.timer = c.regrow_min + Math.floor(this.u() * c.regrow_rand);
      }
    }
    for (let i = 0; i < this.f.food_n; i++) {
      const fd = this.food[i];
      if (fd.on) continue;
      const k = i % this.c.n_meadows;
      fd.timer -= this.fertility(k) * (k === 0 && this.f.n_vents > 0 ? this.c.hollow_rate : 1.0);
      if (fd.timer <= 0) { this.placeFood(i); fd.on = true; this.emit("grow", { index: i, x: fd.x, y: fd.y }); }
    }
    for (const v of this.vents) {
      if (!v.present) continue;
      v.timer -= 1;
      if (v.timer <= 0) { v.on = !v.on; v.timer = this.ventDuration(v.on); }
    }
    if (this.plantTimer > 0) { this.plantTimer -= 1; if (this.plantTimer <= 0) { const g = this.food[c.max_food - 1]; g.on = true; this.emit("grow", { index: c.max_food - 1, x: g.x, y: g.y }); } }
    this.weatherTimer -= 1;
    if (this.weatherTimer <= 0) { this.raining = !this.raining; this.weatherTimer = this.weatherDuration(this.raining); this.emit(this.raining ? "rain_start" : "rain_end"); }
    for (const sh of this.shelters) {  // thickets: dry spells wilt, rain restores; raised thickets grow, live, wither
      if (!sh.on) continue;
      if (sh.life < 0) sh.h = clamp(sh.h + (this.raining ? c.thicket_rain_rate : -c.thicket_dry_rate), 0.0, 1.0);
      else {
        if (sh.life > 0) { sh.h = Math.min(1.0, sh.h + 1.0 / c.raised_grow); sh.life -= 1; }
        if (sh.life === 0) { sh.h = sh.h - 1.0 / c.raised_wither; if (sh.h <= 0) { sh.on = false; sh.h = 0; sh.life = -1; this.emit("thicket_gone", { x: sh.x, y: sh.y }); } }
      }
    }
    for (const z of this.fissures) {  // warning → open → closed
      if (z.state === 0) continue;
      z.timer -= 1;
      if (z.timer <= 0) {
        if (z.state === 1) { z.state = 2; z.timer = c.fissure_open_min + Math.floor(this.u() * c.fissure_open_rand); this.emit("fissure_open", { x: z.x, y: z.y }); }
        else { z.state = 0; this.emit("fissure_closed", { x: z.x, y: z.y }); }
      }
    }
    this.inShelter = this.cover(this.x, this.y);
    const prevMode = this.pred.mode;
    const caught = this.predatorStep();
    if (this.pred.on && this.pred.mode !== prevMode) this.emit("predator_mode", { from: MODES[prevMode], to: MODES[this.pred.mode] });
    if (wasSheltered !== this.inShelter) this.emit(this.inShelter ? "enter_shelter" : "leave_shelter");
    this.perturb();
    this.t += 1; this.lastAction = a;
    const starve = this.energy <= 0;
    this.energy = Math.max(this.energy, 0.0);
    const dead = starve || caught;
    reward += caught ? c.reward_caught : starve ? c.reward_starve : 0.0;
    this.dead = dead; this.caught = caught; this.truncated = !dead && this.t >= c.max_ticks;
    if (caught) this.emit("caught"); else if (starve) this.emit("starved");
    this.updateMemory();
    return { reward, dead, caught, truncated: this.truncated, dashing };
  }

  predatorPercept() {  // seen on any ray within light-limited sight, or heard (same rule as Python)
    const c = this.c, p = this.pred, light = this.light();
    const sight = c.pred_sight_night + (c.pred_sight_day - c.pred_sight_night) * light, r2 = c.predator_radius * c.predator_radius;
    let seen = false;
    if (p.on) for (let k = 0; k < c.n_rays; k++) {
      const ang = this.heading + RAY_REL[k], dx = Math.cos(ang), dy = Math.sin(ang), ox = p.x - this.x, oy = p.y - this.y;
      const proj = ox * dx + oy * dy, perp2 = ox * ox + oy * oy - proj * proj;
      if (!(perp2 <= r2)) continue;
      const thc = Math.sqrt(Math.max(r2 - perp2, 0.0)), t0 = proj - thc, t1 = proj + thc, t = t0 >= 0 ? t0 : t1 >= 0 ? 0.0 : Infinity;
      if (t <= sight) { seen = true; break; }
    }
    let hr = p.mode === HUNT ? (p.pounce ? c.hear_hunt : c.hear_stalk) : c.hear_calm;
    if (this.raining) hr = hr * c.rain_hear;
    const heard = p.on && p.mode !== REST && norm2(p.x - this.x, p.y - this.y) < hr;
    return { seen, heard };
  }
  updateMemory() {
    const { seen, heard } = this.predatorPercept();
    if (seen || heard) { this.memT = 0; this.memX = this.pred.x; this.memY = this.pred.y; this.memPounce = this.pred.pounce && this.pred.mode === HUNT; }
    else this.memT += 1;
  }
  memoryFeatures() {
    const c = this.c, rec = Math.exp(-Math.min(this.memT, 100000) / c.memory_tau);
    const rx = this.memX - this.x, ry = this.memY - this.y, b = Math.atan2(ry, rx) - this.heading;
    const prox = Math.max(0.0, 1.0 - norm2(rx, ry) / c.memory_range);
    return [rec, Math.sin(b) * rec, Math.cos(b) * rec, prox * rec, (this.memPounce ? 1 : 0) * rec];
  }

  observe() {
    const c = this.c, R = c.n_rays, range = c.ray_range;
    const out = new Float64Array(OBS_DIM);
    const px = this.x, py = this.y, light = this.light(), p = this.pred;
    const circle = (best, ox, oy, r, dx, dy) => {
      ox -= px; oy -= py;
      const proj = ox * dx + oy * dy, d2 = ox * ox + oy * oy, perp2 = d2 - proj * proj, r2 = r * r;
      if (!(perp2 <= r2)) return best;
      const thc = Math.sqrt(Math.max(r2 - perp2, 0.0)), t0 = proj - thc, t1 = proj + thc;
      const t = t0 >= 0 ? t0 : t1 >= 0 ? 0.0 : Infinity;
      return t < best ? t : best;
    };
    const prox = (t) => 1.0 - Math.min(t, range) / range;
    const sight = c.pred_sight_night + (c.pred_sight_day - c.pred_sight_night) * light;
    for (let k = 0; k < R; k++) {
      const ang = this.heading + RAY_REL[k], dx = Math.cos(ang), dy = Math.sin(ang);
      const tx = dx > 1e-12 ? (c.width - px) / dx : dx < -1e-12 ? (0.0 - px) / dx : Infinity;
      const ty = dy > 1e-12 ? (c.height - py) / dy : dy < -1e-12 ? (0.0 - py) / dy : Infinity;
      let ts = Infinity, tf = Infinity, th = Infinity, tsh = Infinity, tp = Infinity, tw = Infinity;
      for (const o of this.slates) if (o.on) ts = circle(ts, o.x, o.y, o.r, dx, dy);
      ts = Math.min(Math.min(tx, ty), ts);
      for (const fd of this.food) if (fd.on) tf = circle(tf, fd.x, fd.y, c.food_radius, dx, dy);
      for (const v of this.vents) if (v.on) th = circle(th, v.x, v.y, v.r, dx, dy);
      let vHaz = prox(th), vSh = 0;
      for (const z of this.fissures) if (z.state > 0) { const t = circle(Infinity, z.x, z.y, c.fissure_radius, dx, dy); vHaz = Math.max(vHaz, prox(t) * (z.state === 2 ? 1.0 : c.fissure_warn_signal)); }
      for (const s of this.shelters) if (s.on) { const t = circle(Infinity, s.x, s.y, s.r, dx, dy); vSh = Math.max(vSh, prox(t) * s.h); }
      if (p.on) { tp = circle(tp, p.x, p.y, c.predator_radius, dx, dy); if (!(tp <= sight)) tp = Infinity; }
      for (const q of this.pools) if (q.on) tw = circle(tw, q.x, q.y, q.r, dx, dy);
      const b = N_CH * k;
      out[b] = prox(ts); out[b + 1] = prox(tf); out[b + 2] = vHaz; out[b + 3] = vSh; out[b + 4] = prox(tp); out[b + 5] = prox(tw);
    }
    let hr = p.mode === HUNT ? (p.pounce ? c.hear_hunt : c.hear_stalk) : c.hear_calm;
    if (this.raining) hr = hr * c.rain_hear;
    const rx = p.x - px, ry = p.y - py, dpr = norm2(rx, ry);
    const heard = p.on && p.mode !== REST && dpr < hr;
    const bearing = Math.atan2(ry, rx) - this.heading;
    const phase = (TWO_PI * (this.t + this.t0)) / c.day_ticks;
    const S = N_CH * R;
    const selfv = [this.energy, this.stamina, this.v / c.v_max, this.w / c.w_max, this.inShelter ? 1 : 0, this.inHazard ? 1 : 0,
      this.inWater ? 1 : 0, this.contact ? 1 : 0, light, this.raining ? 1 : 0, Math.sin(phase), Math.cos(phase),
      heard ? 1.0 - dpr / hr : 0.0, heard ? Math.sin(bearing) : 0.0, heard ? Math.cos(bearing) : 0.0, this.injury / c.injury_ticks];
    for (let i = 0; i < selfv.length; i++) out[S + i] = selfv[i];
    const mem = this.memoryFeatures();
    for (let i = 0; i < N_MEM; i++) out[S + selfv.length + i] = mem[i];
    out[S + selfv.length + N_MEM + this.lastAction] = 1;
    return out;
  }

  snapshot() {
    const on = (a) => a.filter((e) => e.on).map((e) => ({ ...e }));
    return { t: this.t, x: this.x, y: this.y, heading: this.heading, v: this.v, w: this.w, energy: this.energy, stamina: this.stamina,
      contact: this.contact, inHazard: this.inHazard, inShelter: this.inShelter, inWater: this.inWater, resting: this.resting,
      light: this.light(), raining: this.raining, slates: on(this.slates), shelters: on(this.shelters), pools: on(this.pools),
      vents: this.vents.filter((v) => v.present).map((v) => ({ ...v })), food: on(this.food), meadows: this.meadows.map((m, k) => ({ ...m, fertility: this.fertility(k) })),
      fissures: this.fissures.filter((z) => z.state > 0).map((z) => ({ ...z })), plantTimer: this.plantTimer, injury: this.injury,
      pred: { ...this.pred, mode: MODES[this.pred.mode] }, epFood: this.epFood, dead: this.dead, caught: this.caught };
  }
}
