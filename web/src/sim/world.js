// GENESIS world v1 — single-env JS port of genesis/world.py. Same constants, same PRNG draw order.
// See docs/WORLD_SPEC.md. Float64 throughout. Do not use Math.random or Math.hypot here.
import C from "./world_constants.json" with { type: "json" };

export const SPEC_VERSION = C.world_spec_version;
export const N_ACTIONS = C.actions.length;
export const OBS_DIM = C.n_rays * 3 + 3 + N_ACTIONS;
export const CONSTANTS = C;
const TWO_PI = 2 * Math.PI;

export class Mulberry32 {
  constructor(seed) { this.s = seed >>> 0; }
  next() {
    this.s = (this.s + 0x6D2B79F5) >>> 0;
    let s = this.s;
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
const RAY_REL = rayAngles();
const norm2 = (x, y) => Math.sqrt(x * x + y * y);

export class World {
  constructor(family = "standard", consts = C) {
    this.c = consts; this.f = { ...consts.families[family] }; this.family = family;
    this.events = [];
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

  obstaclesOk(x, y, extra) {
    for (const o of this.slates) if (o.on && !(norm2(o.x - x, o.y - y) > o.r + extra)) return false;
    return true;
  }
  notInVents(x, y) {
    for (const v of this.vents) if (v.on && !(norm2(v.x - x, v.y - y) > v.r)) return false;
    return true;
  }
  ventDuration(active) {
    const u = this.u();
    return active ? 150 + Math.floor(u * 250) : 100 + Math.floor(u * 200);
  }
  placeFood(i) {
    const rf = this.c.food_radius;
    const [x, y] = this.samplePoint(0.4, (x, y) =>
      this.obstaclesOk(x, y, rf + 0.1) && this.notInVents(x, y) && norm2(x - this.x, y - this.y) > 1.0);
    this.food[i].x = x; this.food[i].y = y;
  }

  reset(seed) {
    const c = this.c, f = this.f, rc = c.creature_radius;
    this.seed = seed;
    this.rng = new Mulberry32(seed);
    this.slates = Array.from({ length: c.max_obstacles }, () => ({ x: 0, y: 0, r: 0, on: false }));
    this.vents = Array.from({ length: c.max_vents }, () => ({ x: -100, y: -100, r: 0, on: false, present: false, timer: 0 }));
    this.food = Array.from({ length: c.max_food }, () => ({ x: -100, y: -100, on: false }));
    this.x = -100; this.y = -100;
    const nObs = f.obs_min + Math.floor(this.u() * (f.obs_max - f.obs_min + 1));
    for (let j = 0; j < c.max_obstacles; j++) {
      if (j >= nObs) continue;
      const r = f.obs_rmin + this.u() * (f.obs_rmax - f.obs_rmin);
      const gap = 2 * rc + 0.05;
      const [x, y] = this.samplePoint(r + gap, (x, y) => this.obstaclesOk(x, y, r + gap));
      Object.assign(this.slates[j], { x, y, r, on: true });
    }
    for (let v = 0; v < c.max_vents; v++) {
      if (v >= f.n_vents) continue;
      const r = 0.6 + this.u() * 0.4;
      const [x, y] = this.samplePoint(r, (x, y) => this.obstaclesOk(x, y, 0.0));
      const active = this.u() < 0.7;
      Object.assign(this.vents[v], { x, y, r, present: true, on: active });
      this.vents[v].timer = this.ventDuration(active);
    }
    const [cx, cy] = this.samplePoint(rc + 0.1, (x, y) => {
      if (!this.obstaclesOk(x, y, rc + 0.1)) return false;
      for (const v of this.vents) if (v.present && !(norm2(v.x - x, v.y - y) > v.r + rc)) return false;
      return true;
    });
    this.x = cx; this.y = cy;
    this.heading = this.u() * TWO_PI;
    for (let i = 0; i < f.food_n; i++) { this.placeFood(i); this.food[i].on = true; }
    this.energy = c.energy_start; this.t = 0; this.lastAction = 5;
    this.contact = false; this.inHazard = false;
    this.epFood = 0; this.dead = false; this.truncated = false;
    this.events = [];
  }

  // ---- visitor / perturbation mutations (same clearance rules as Python)
  insertObstacle(r = null, at = null) {
    const c = this.c, f = this.f, rc = c.creature_radius;
    if (this.slates.every((o) => o.on)) return false;
    if (r === null) r = f.obs_rmin + this.u() * (f.obs_rmax - f.obs_rmin);
    const clear = (x, y) => {
      if (!this.obstaclesOk(x, y, r + 2 * rc + 0.05)) return false;
      if (!(norm2(x - this.x, y - this.y) > r + rc + 0.5)) return false;
      for (const fd of this.food) if (fd.on && !(norm2(fd.x - x, fd.y - y) > r + c.food_radius)) return false;
      return true;
    };
    let x, y, ok;
    if (at === null) [x, y, ok] = this.samplePoint(r + 2 * rc + 0.05, clear);
    else { [x, y] = at; ok = clear(x, y); }
    if (!ok) return false;
    const slot = this.slates.findIndex((o) => !o.on);
    Object.assign(this.slates[slot], { x, y, r, on: true });
    return true;
  }
  wakeVent(v) {
    if (!this.vents[v].present) return false;
    this.vents[v].on = true; this.vents[v].timer = 300; return true;
  }
  moveFood(i, x, y) {  // visitor seed placement: same validity rule as the sampler
    const rf = this.c.food_radius, m = 0.4;
    if (!this.food[i].on || x < m || y < m || x > this.c.width - m || y > this.c.height - m) return false;
    if (!(this.obstaclesOk(x, y, rf + 0.1) && this.notInVents(x, y) && norm2(x - this.x, y - this.y) > 1.0)) return false;
    this.food[i].x = x; this.food[i].y = y; return true;
  }
  perturb() {
    const f = this.f;
    const u = this.u();
    if (u < f.p_food) {
      const k = Math.min(Math.floor(this.u() * f.food_n), f.food_n - 1);
      this.placeFood(k);
    } else if (u < f.p_food + f.p_obs) {
      this.insertObstacle();
    } else if (u < f.p_food + f.p_obs + f.p_vent && f.n_vents > 0) {
      const k = Math.min(Math.floor(this.u() * f.n_vents), f.n_vents - 1);
      this.wakeVent(k);
    }
  }

  step(a) {
    const c = this.c, rc = c.creature_radius;
    const dturn = a === 1 ? -c.turn_fwd : a === 2 ? c.turn_fwd : a === 3 ? -c.turn_inplace : a === 4 ? c.turn_inplace : 0.0;
    const spd = a === 0 ? c.speed : a === 1 || a === 2 ? c.speed_turn_fwd : 0.0;
    let h = (this.heading + dturn) % TWO_PI; if (h < 0) h += TWO_PI;
    this.heading = h;
    let px = this.x + spd * Math.cos(h), py = this.y + spd * Math.sin(h);
    let contact = false;
    for (const o of this.slates) {
      const dx = px - o.x, dy = py - o.y, d = norm2(dx, dy), minD = rc + o.r;
      if (o.on && d < minD) {
        const nx = d > 1e-12 ? dx / d : 1.0, ny = d > 1e-12 ? dy / d : 0.0;
        px = o.x + nx * minD; py = o.y + ny * minD; contact = true;
      }
    }
    const qx = Math.min(Math.max(px, rc), c.width - rc), qy = Math.min(Math.max(py, rc), c.height - rc);
    if (qx !== px || qy !== py) contact = true;
    this.x = qx; this.y = qy; this.contact = contact;
    let reward = 0;
    this.energy -= c.energy_decay + c.move_cost * (spd > 0 ? 1 : 0);
    let inh = false;
    for (const v of this.vents) if (v.on && norm2(v.x - this.x, v.y - this.y) < v.r) inh = true;
    this.inHazard = inh;
    if (inh) { this.energy -= c.hazard_drain; reward += c.reward_hazard_tick; }
    for (let i = 0; i < c.max_food; i++) {
      const fd = this.food[i];
      if (fd.on && norm2(fd.x - this.x, fd.y - this.y) < rc + c.food_radius) {
        this.energy = Math.min(1.0, this.energy + c.food_gain);
        reward += c.reward_food; this.epFood += 1;
        this.events.push({ tick: this.t, kind: "eat", index: i, id: `${this.seed}:${this.t}:${i}` });
        this.placeFood(i);
      }
    }
    for (const v of this.vents) {
      if (!v.present) continue;
      v.timer -= 1;
      if (v.timer <= 0) { v.on = !v.on; v.timer = this.ventDuration(v.on); }
    }
    this.perturb();
    this.t += 1; this.lastAction = a;
    const dead = this.energy <= 0;
    this.energy = Math.max(this.energy, 0.0);
    if (dead) reward += c.reward_death;
    this.dead = dead; this.truncated = !dead && this.t >= c.max_ticks;
    return { reward, dead, truncated: this.truncated };
  }

  observe() {
    const c = this.c, R = c.n_rays, range = c.ray_range;
    const out = new Float64Array(OBS_DIM);
    const px = this.x, py = this.y;
    const circles = (list, dx, dy, radiusOf, onOf) => {
      let best = Infinity;
      for (const e of list) {
        if (!onOf(e)) continue;
        const ox = e.x - px, oy = e.y - py, r = radiusOf(e);
        const proj = ox * dx + oy * dy, d2 = ox * ox + oy * oy, perp2 = d2 - proj * proj, r2 = r * r;
        if (!(perp2 <= r2)) continue;
        const thc = Math.sqrt(Math.max(r2 - perp2, 0.0));
        const t0 = proj - thc, t1 = proj + thc;
        const t = t0 >= 0 ? t0 : t1 >= 0 ? 0.0 : Infinity;
        if (t < best) best = t;
      }
      return best;
    };
    const prox = (t) => 1.0 - Math.min(t, range) / range;
    for (let k = 0; k < R; k++) {
      const ang = this.heading + RAY_REL[k];
      const dx = Math.cos(ang), dy = Math.sin(ang);
      const tx = dx > 1e-12 ? (c.width - px) / dx : dx < -1e-12 ? (0.0 - px) / dx : Infinity;
      const ty = dy > 1e-12 ? (c.height - py) / dy : dy < -1e-12 ? (0.0 - py) / dy : Infinity;
      const tSolid = Math.min(Math.min(tx, ty), circles(this.slates, dx, dy, (o) => o.r, (o) => o.on));
      const tFood = circles(this.food, dx, dy, () => c.food_radius, (f) => f.on);
      const tHaz = circles(this.vents, dx, dy, (v) => v.r, (v) => v.on);
      out[3 * k] = prox(tSolid); out[3 * k + 1] = prox(tFood); out[3 * k + 2] = prox(tHaz);
    }
    out[3 * R] = this.energy; out[3 * R + 1] = this.inHazard ? 1 : 0; out[3 * R + 2] = this.contact ? 1 : 0;
    out[3 * R + 3 + this.lastAction] = 1;
    return out;
  }

  snapshot() {
    return { t: this.t, x: this.x, y: this.y, heading: this.heading, energy: this.energy, contact: this.contact,
      inHazard: this.inHazard, slates: this.slates.filter((o) => o.on).map((o) => ({ ...o })),
      vents: this.vents.filter((v) => v.present).map((v) => ({ ...v })),
      food: this.food.filter((f) => f.on).map((f) => ({ ...f })), epFood: this.epFood, dead: this.dead };
  }
}
