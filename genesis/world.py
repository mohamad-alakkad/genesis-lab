"""GENESIS world v2 — batch-vectorized NumPy reference implementation (float64).

v2 adds momentum, stamina, rest, shelters, meadows, a day/night cycle and a predator whose
behavior is a scripted world rule. The creature's decisions come only from the learned policy.
All world randomness comes from a per-env mulberry32 stream (portable to JS); the JS port in
web/src/sim/world.js follows exactly the same draw order (see docs/WORLD_SPEC.md §7).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

C = json.loads((Path(__file__).with_name("world_constants.json")).read_text())
SPEC_VERSION = C["world_spec_version"]
N_ACTIONS = len(C["actions"])
N_CH = 6  # ray channels: solid, seed, vent, shelter, predator, water
N_MEM = 5  # memory of Vela's own last predator percept (2.2.0)
N_SELF = 16  # 3.0.0: + injury
OBS_DIM = C["n_rays"] * N_CH + N_SELF + N_MEM + N_ACTIONS
U32 = np.uint32
TWO_PI = 2 * math.pi
PATROL, HUNT, SEARCH, REST = 0, 1, 2, 3
A_DASH, A_REST = 3, 7


class Mulberry32:
    """Vectorized mulberry32; next(mask) advances only masked streams."""

    def __init__(self, n: int):
        self.s = np.zeros(n, dtype=U32)

    def seed(self, mask: np.ndarray, seeds: np.ndarray):
        self.s[mask] = np.asarray(seeds, dtype=np.uint64).astype(U32)

    def next(self, mask: np.ndarray) -> np.ndarray:
        with np.errstate(over="ignore"):
            s = self.s + U32(0x6D2B79F5)
            self.s = np.where(mask, s, self.s)
            t = (s ^ (s >> U32(15))) * (s | U32(1))
            t = (t + ((t ^ (t >> U32(7))) * (t | U32(61)))) ^ t
            r = (t ^ (t >> U32(14))).astype(np.float64) / 4294967296.0
        return np.where(mask, r, 0.0)


def _norm(v):
    return np.sqrt(v[..., 0] * v[..., 0] + v[..., 1] * v[..., 1])


class World:
    def __init__(self, n_envs: int, family: str = "train", consts: dict | None = None):
        self.c = c = dict(consts or C)
        self.f = dict(c["families"][family])
        self.family = family
        E, MO, MF, MV, MS, NM = n_envs, c["max_obstacles"], c["max_food"], c["max_vents"], c["max_shelters"], c["n_meadows"]
        self.E = E
        self.rng = Mulberry32(E)
        z = lambda *s: np.zeros((E, *s))
        self.pos = z(2); self.heading = z(); self.v = z(); self.w = z(); self.energy = z(); self.stamina = z()
        self.t = np.zeros(E, np.int64); self.t0 = np.zeros(E, np.int64); self.last_action = np.full(E, A_REST, np.int64)
        self.contact = np.zeros(E, bool); self.in_hazard = np.zeros(E, bool); self.in_shelter = np.zeros(E, bool)
        self.resting = np.zeros(E, bool); self.in_water = np.zeros(E, bool)
        MP = c["max_pools"]
        self.pool_pos = z(MP, 2); self.pool_r = z(MP); self.pool_on = np.zeros((E, MP), bool)
        self.raining = np.zeros(E, bool); self.weather_timer = np.zeros(E, np.int64)
        self.obs_pos = z(MO, 2); self.obs_r = z(MO); self.obs_on = np.zeros((E, MO), bool)
        self.sh_pos = z(MS, 2); self.sh_r = z(MS); self.sh_on = np.zeros((E, MS), bool)
        self.vent_pos = z(MV, 2); self.vent_r = z(MV); self.vent_on = np.zeros((E, MV), bool)
        self.vent_present = np.zeros((E, MV), bool); self.vent_timer = np.zeros((E, MV), np.int64)
        self.meadow = z(NM, 2)
        self.food_pos = z(MF, 2); self.food_on = np.zeros((E, MF), bool); self.food_timer = z(MF)
        self.plant_timer = np.zeros(E, np.int64)  # ripening of the planted seed (slot max_food-1)
        self.meadow_phase = np.zeros((E, NM), np.int64)
        self.sh_h = z(MS); self.sh_life = np.full((E, MS), -1, np.int64)  # thicket health; life ≥ 0 = raised thicket
        MFZ = c["max_fissures"]
        self.fz_pos = z(MFZ, 2); self.fz_state = np.zeros((E, MFZ), np.int64); self.fz_timer = np.zeros((E, MFZ), np.int64)
        self.injury = np.zeros(E, np.int64)
        self.pr_on = np.zeros(E, bool); self.pr_pos = z(2); self.pr_heading = z(); self.pr_mode = np.zeros(E, np.int64)
        self.pr_timer = np.zeros(E, np.int64); self.pr_fatigue = z(); self.pr_target = z(2); self.pr_last = z(2)
        self.pr_speed = z(); self.pr_seen = np.zeros(E, bool); self.pr_pounce = np.zeros(E, bool)
        self.mem_t = np.full(E, 10**6, np.int64); self.mem_pos = z(2); self.mem_pounce = np.zeros(E, bool)
        self.stat_keys = ("food", "hazard", "dist", "contact", "ret", "caught", "threat", "shelter_threat", "shelter",
                          "rest", "dash", "dash_threat", "night", "shelter_night", "injury")
        self.ep = {k: z() for k in self.stat_keys}
        self.events = []
        self.ray_rel = np.linspace(-c["fov"] / 2, c["fov"] / 2, c["n_rays"])

    # ----------------------------------------------------------------- helpers
    def _u(self, mask):
        return self.rng.next(mask)

    def _sample_point(self, mask, margin, clear_fn):
        W, H = self.c["width"], self.c["height"]
        need = mask.copy(); out = np.zeros((self.E, 2)); ok = np.zeros(self.E, bool)
        margin = np.broadcast_to(np.asarray(margin, np.float64), (self.E,))
        for _ in range(self.c["sample_attempts"]):
            if not need.any():
                break
            x = margin + self._u(need) * (W - 2 * margin)
            y = margin + self._u(need) * (H - 2 * margin)
            p = np.stack([x, y], 1)
            out[need] = p[need]
            good = need & clear_fn(p)
            ok |= good; need &= ~good
        return out, ok

    def _sample_polar(self, mask, center, R, clear_fn):
        """Seeds grow around a meadow: angle draw then radius draw; out-of-bounds candidates rejected."""
        W, H = self.c["width"], self.c["height"]
        need = mask.copy(); out = np.zeros((self.E, 2))
        for _ in range(self.c["sample_attempts"]):
            if not need.any():
                break
            a = self._u(need) * TWO_PI
            r = R * np.sqrt(self._u(need))
            p = center + np.stack([r * np.cos(a), r * np.sin(a)], 1)
            out[need] = p[need]
            inb = (p[:, 0] >= 0.4) & (p[:, 0] <= W - 0.4) & (p[:, 1] >= 0.4) & (p[:, 1] <= H - 0.4)
            good = need & inb & clear_fn(p)
            need &= ~good
        out[:, 0] = np.clip(out[:, 0], 0.4, W - 0.4); out[:, 1] = np.clip(out[:, 1], 0.4, H - 0.4)
        return out

    @staticmethod
    def _extra(ex):
        ex = np.asarray(ex, np.float64)
        return ex[:, None] if ex.ndim == 1 else ex

    def _slates_ok(self, p, extra):
        d = _norm(self.obs_pos - p[:, None, :])
        return np.all(~self.obs_on | (d > self.obs_r + self._extra(extra)), 1)

    def _shelters_ok(self, p, extra):
        d = _norm(self.sh_pos - p[:, None, :])
        return np.all(~self.sh_on | (d > self.sh_r + self._extra(extra)), 1)

    def _pools_ok(self, p, extra):
        d = _norm(self.pool_pos - p[:, None, :])
        return np.all(~self.pool_on | (d > self.pool_r + self._extra(extra)), 1)

    def _in_pool(self, p):
        return ~self._pools_ok(p, 0.0)

    def _weather_duration(self, mask, rain):
        u = self._u(mask)
        c = self.c
        return np.where(rain, c["rain_min"] + np.floor(u * c["rain_rand"]),
                        c["dry_min"] + np.floor(u * c["dry_rand"])).astype(np.int64)

    def _not_in_vents(self, p):
        d = _norm(self.vent_pos - p[:, None, :])
        return np.all(~self.vent_on | (d > self.vent_r), 1)

    def _vent_duration(self, mask, active):
        u = self._u(mask)
        return np.where(active, 150 + np.floor(u * 250), 100 + np.floor(u * 200)).astype(np.int64)

    def _place_food(self, mask, i):
        rf = self.c["food_radius"]
        m = i % self.c["n_meadows"]
        def clear(p):
            return (self._slates_ok(p, rf + 0.1) & self._not_in_vents(p) & self._shelters_ok(p, rf)
                    & self._pools_ok(p, rf) & (_norm(p - self.pos) > 1.0))
        pts = self._sample_polar(mask, self.meadow[:, m], self.c["meadow_spread"], clear)
        self.food_pos[mask, i] = pts[mask]

    def _new_waypoint(self, mask):
        pr = self.c["predator_radius"]
        pts, _ = self._sample_point(mask, 0.5, lambda p: self._slates_ok(p, pr + 0.1) & self._shelters_ok(p, pr + 0.1))
        self.pr_target[mask] = pts[mask]

    def light(self, t=None):
        t = self.t if t is None else t
        return 0.5 + 0.5 * np.cos(TWO_PI * (t + self.t0) / self.c["day_ticks"])

    # ----------------------------------------------------------------- reset
    def reset(self, mask: np.ndarray, seeds: np.ndarray):
        c, f = self.c, self.f
        m = np.asarray(mask, bool)
        if not m.any():
            return
        self.rng.seed(m, np.asarray(seeds)[m] if np.size(seeds) == self.E else seeds)
        rc, pr = c["creature_radius"], c["predator_radius"]
        gap = 2 * rc + 0.05
        for a in (self.obs_on, self.sh_on, self.vent_on, self.vent_present, self.food_on, self.pool_on):
            a[m] = False
        for a in (self.obs_r, self.sh_r, self.vent_r, self.pool_r):
            a[m] = 0.0
        for a in (self.obs_pos, self.sh_pos, self.vent_pos, self.meadow, self.food_pos, self.pool_pos):
            a[m] = -100.0
        self.pos[m] = -100.0; self.pr_pos[m] = -100.0; self.pr_on[m] = False
        # 1. slates
        n = f["obs_min"] + np.floor(self._u(m) * (f["obs_max"] - f["obs_min"] + 1)).astype(np.int64)
        for j in range(c["max_obstacles"]):
            mj = m & (j < n)
            if not mj.any():
                continue
            r = f["obs_rmin"] + self._u(mj) * (f["obs_rmax"] - f["obs_rmin"])
            pts, _ = self._sample_point(mj, r + gap, lambda p: self._slates_ok(p, r + gap))
            self.obs_pos[mj, j] = pts[mj]; self.obs_r[mj, j] = r[mj]; self.obs_on[mj, j] = True
        # 2. shelters
        n = f["sh_min"] + np.floor(self._u(m) * (f["sh_max"] - f["sh_min"] + 1)).astype(np.int64)
        for s in range(c["max_shelters"]):
            ms = m & (s < n)
            if not ms.any():
                continue
            r = 0.9 + self._u(ms) * 0.4
            wall = r + 2 * pr + 0.05  # 3.0.0: the predator can always pass between a thicket and a wall
            pts, _ = self._sample_point(ms, wall, lambda p: self._slates_ok(p, r + gap) & self._shelters_ok(p, r + 1.0))
            self.sh_pos[ms, s] = pts[ms]; self.sh_r[ms, s] = r[ms]; self.sh_on[ms, s] = True
        # 2b. pools (shallow water: slows both creature and predator)
        n = f["pool_min"] + np.floor(self._u(m) * (f["pool_max"] - f["pool_min"] + 1)).astype(np.int64)
        for q in range(c["max_pools"]):
            mq = m & (q < n)
            if not mq.any():
                continue
            r = 1.2 + self._u(mq) * 0.8
            pts, _ = self._sample_point(mq, r + 0.3, lambda p: self._slates_ok(p, r + gap) & self._shelters_ok(p, r + 0.8)
                                        & self._pools_ok(p, r + 1.0))
            self.pool_pos[mq, q] = pts[mq]; self.pool_r[mq, q] = r[mq]; self.pool_on[mq, q] = True
        # 3. vents
        for v in range(c["max_vents"]):
            mv = m & (v < f["n_vents"])
            if not mv.any():
                continue
            r = 0.6 + self._u(mv) * 0.3
            pts, _ = self._sample_point(mv, r, lambda p: self._slates_ok(p, 0.0) & self._shelters_ok(p, r + 0.5)
                                        & self._pools_ok(p, r))
            active = self._u(mv) < 0.7
            self.vent_pos[mv, v] = pts[mv]; self.vent_r[mv, v] = r[mv]; self.vent_present[mv, v] = True
            self.vent_on[mv, v] = active[mv]
            self.vent_timer[mv, v] = self._vent_duration(mv, active)[mv]
        # 4. meadows: open ground at a thicket's edge (food in the open, cover within reach)
        def clear_m(p):
            d = _norm(self.sh_pos - p[:, None, :]) - self.sh_r
            near = np.any(self.sh_on & (d <= c["meadow_shelter_max"]), 1)
            return self._shelters_ok(p, c["meadow_shelter_min"]) & near & self._slates_ok(p, 0.0) & self._pools_ok(p, 1.0)
        def clear_hollow(p):  # meadow 0: the rich hollow beside the first vent (no cover nearby)
            dv = _norm(self.vent_pos[:, 0] - p) - self.vent_r[:, 0]
            return ((dv >= c["hollow_vent_min"]) & (dv <= c["hollow_vent_max"]) & self._shelters_ok(p, 1.0)
                    & self._slates_ok(p, 0.0) & self._pools_ok(p, 1.0))
        for k in range(c["n_meadows"]):
            pts, _ = self._sample_point(m, 1.5, clear_hollow if (k == 0 and f["n_vents"] > 0) else clear_m)
            self.meadow[m, k] = pts[m]
        # 5. creature
        def clear_c(p):
            dv = _norm(self.vent_pos - p[:, None, :])
            return self._slates_ok(p, rc + 0.1) & np.all(~self.vent_present | (dv > self.vent_r + rc), 1)
        pts, _ = self._sample_point(m, rc + 0.1, clear_c)
        self.pos[m] = pts[m]
        self.heading[m] = self._u(m)[m] * TWO_PI
        # 6. predator
        if f["predator"]:
            pts, _ = self._sample_point(m, 0.5, lambda p: self._slates_ok(p, pr + 0.1) & self._shelters_ok(p, pr + 0.1)
                                        & (_norm(p - self.pos) > 8.0))
            self.pr_pos[m] = pts[m]; self.pr_on[m] = True
            self.pr_speed[m] = (f["hunt_min"] + self._u(m) * (f["hunt_max"] - f["hunt_min"]))[m]
            self._new_waypoint(m)
            self.pr_mode[m] = PATROL; self.pr_timer[m] = 0; self.pr_fatigue[m] = 0.0
            self.pr_heading[m] = 0.0; self.pr_last[m] = self.pr_pos[m]
        self.pr_seen[m] = False; self.pr_pounce[m] = False
        # 7. seeds
        for i in range(f["food_n"]):
            self._place_food(m, i)
            self.food_on[m, i] = True
        self.food_timer[m] = 0.0; self.plant_timer[m] = 0
        # 8. time of day
        if f["t0_fixed"] >= 0:
            self.t0[m] = f["t0_fixed"]
        else:
            self.t0[m] = np.floor(self._u(m) * c["day_ticks"]).astype(np.int64)[m]
        # 9. weather
        rain = self._u(m) < c["rain_start_p"]
        self.raining[m] = rain[m]
        self.weather_timer[m] = self._weather_duration(m, rain)[m]
        self.in_water[m] = self._in_pool(self.pos)[m]
        if not f["rain"]:  # drought family: the sky stays dry (draws above still consumed for comparability)
            self.raining[m] = False; self.weather_timer[m] = 10**9
        # 10. meadow fertility phases · 11. thicket health · fissures closed · no injury
        for k in range(c["n_meadows"]):
            self.meadow_phase[m, k] = np.floor(self._u(m) * c["fertility_period"]).astype(np.int64)[m]
        for sl in range(c["max_shelters"]):
            ms = m & self.sh_on[:, sl]
            if ms.any():
                self.sh_h[ms, sl] = (c["thicket_h_min"] + self._u(ms) * (1 - c["thicket_h_min"]))[ms]
        self.sh_life[m] = -1
        self.fz_state[m] = 0; self.fz_timer[m] = 0; self.fz_pos[m] = -100.0
        self.injury[m] = 0
        self.energy[m] = c["energy_start"]; self.stamina[m] = 1.0; self.v[m] = 0.0; self.w[m] = 0.0
        self.t[m] = 0; self.last_action[m] = A_REST
        self.contact[m] = False; self.in_hazard[m] = False; self.resting[m] = False
        self.in_shelter[m] = self._cover(self.pos)[m]
        self.mem_t[m] = 10**6; self.mem_pos[m] = 0.0; self.mem_pounce[m] = False
        self._update_memory(m)
        for k in self.stat_keys:
            self.ep[k][m] = 0.0

    # ----------------------------------------------------------------- visitor / perturbation mutations
    def insert_obstacle(self, mask, r=None, at=None):
        c, f = self.c, self.f
        rc, pr = c["creature_radius"], c["predator_radius"]
        mask = mask & ~np.all(self.obs_on, 1)
        if not mask.any():
            return np.zeros(self.E, bool)
        if r is None:
            r = f["obs_rmin"] + self._u(mask) * (f["obs_rmax"] - f["obs_rmin"])
        r = np.broadcast_to(np.asarray(r, np.float64), (self.E,))
        def clear(p):
            dfood = _norm(self.food_pos - p[:, None, :])
            return (self._slates_ok(p, r + 2 * rc + 0.05) & self._shelters_ok(p, r + 0.2) & self._pools_ok(p, r)
                    & (_norm(p - self.pos) > r + rc + 0.5) & (~self.pr_on | (_norm(p - self.pr_pos) > r + pr + 0.2))
                    & np.all(~self.food_on | (dfood > r[:, None] + c["food_radius"]), 1))
        if at is None:
            pts, ok = self._sample_point(mask, r + 2 * rc + 0.05, clear)
        else:
            pts = np.broadcast_to(np.asarray(at, np.float64), (self.E, 2)).copy()
            W, H = c["width"], c["height"]
            inb = (pts[:, 0] >= r) & (pts[:, 1] >= r) & (pts[:, 0] <= W - r) & (pts[:, 1] <= H - r)
            ok = mask & inb & clear(pts)
        ok &= mask
        slot = np.argmin(self.obs_on, 1)
        idx = np.nonzero(ok)[0]
        self.obs_pos[idx, slot[idx]] = pts[idx]; self.obs_r[idx, slot[idx]] = r[idx]; self.obs_on[idx, slot[idx]] = True
        return ok

    def move_food(self, mask, i, xy):
        c = self.c
        p = np.broadcast_to(np.asarray(xy, np.float64), (self.E, 2)).copy()
        mg = 0.4
        ok = (mask & self.food_on[:, i] & (p[:, 0] >= mg) & (p[:, 1] >= mg) & (p[:, 0] <= c["width"] - mg)
              & (p[:, 1] <= c["height"] - mg) & self._slates_ok(p, c["food_radius"] + 0.1)
              & self._not_in_vents(p) & self._shelters_ok(p, c["food_radius"]) & self._pools_ok(p, c["food_radius"])
              & (_norm(p - self.pos) > 1.0))
        self.food_pos[ok, i] = p[ok]
        return ok

    def wake_vent(self, mask, v):
        mm = mask & self.vent_present[:, v]
        self.vent_on[mm, v] = True
        self.vent_timer[mm, v] = 300
        return mm

    def alert_predator(self, mask, at=None):
        """A disturbance draws the predator to search a point (default: where Vela is). A visitor 'lure' chooses the point."""
        mm = mask & self.pr_on & (self.pr_mode != REST)
        pt = self.pos if at is None else np.broadcast_to(np.asarray(at, np.float64), (self.E, 2))
        self.pr_mode[mm] = SEARCH; self.pr_last[mm] = pt[mm]; self.pr_timer[mm] = self.c["predator_search_ticks"]
        return mm

    def _cover(self, p):
        """Effective cover: inside a thicket whose health is above the concealment threshold."""
        d = _norm(self.sh_pos - p[:, None, :])
        return np.any(self.sh_on & (self.sh_h > self.c["cover_min"]) & (d < self.sh_r), 1)

    def fertility(self, k):
        c = self.c
        cyc = 0.5 + 0.5 * np.sin(TWO_PI * (self.t + self.meadow_phase[:, k]) / c["fertility_period"])
        return c["fertility_floor"] + c["fertility_gain"] * cyc + c["rain_fertility"] * self.raining

    def plant_seed(self, mask, at=None):
        """A seed sprouts and ripens over seed_ripen ticks (slot max_food-1)."""
        c = self.c; g = c["max_food"] - 1; rf = c["food_radius"]
        mask = mask & ~self.food_on[:, g] & (self.plant_timer <= 0)
        if not mask.any():
            return np.zeros(self.E, bool)
        def clear(p):
            return (self._slates_ok(p, rf + 0.1) & self._not_in_vents(p) & self._shelters_ok(p, rf) & self._pools_ok(p, rf)
                    & (_norm(p - self.pos) > 1.0))
        if at is None:
            pts, ok = self._sample_point(mask, 0.5, clear)
        else:
            pts = np.broadcast_to(np.asarray(at, np.float64), (self.E, 2)).copy()
            W, H = c["width"], c["height"]
            ok = mask & (pts[:, 0] >= 0.5) & (pts[:, 1] >= 0.5) & (pts[:, 0] <= W - 0.5) & (pts[:, 1] <= H - 0.5) & clear(pts)
        ok &= mask
        self.food_pos[ok, g] = pts[ok]; self.plant_timer[ok] = c["seed_ripen"]
        return ok

    def raise_thicket(self, mask, at=None):
        """A young thicket grows (raised_grow ticks), lives raised_life ticks, then withers (slot max_shelters-1)."""
        c = self.c; sl = c["max_shelters"] - 1; r = c["raised_radius"]
        mask = mask & ~self.sh_on[:, sl]
        if not mask.any():
            return np.zeros(self.E, bool)
        def clear(p):
            return self._slates_ok(p, r + 0.2) & self._shelters_ok(p, r + 0.5) & self._pools_ok(p, r)
        wall = r + 2 * c["predator_radius"] + 0.05
        if at is None:
            pts, ok = self._sample_point(mask, wall, clear)
        else:
            pts = np.broadcast_to(np.asarray(at, np.float64), (self.E, 2)).copy()
            W, H = c["width"], c["height"]
            ok = mask & (pts[:, 0] >= wall) & (pts[:, 1] >= wall) & (pts[:, 0] <= W - wall) & (pts[:, 1] <= H - wall) & clear(pts)
        ok &= mask
        self.sh_pos[ok, sl] = pts[ok]; self.sh_r[ok, sl] = r; self.sh_on[ok, sl] = True
        self.sh_h[ok, sl] = 0.0; self.sh_life[ok, sl] = c["raised_life"]
        return ok

    def call_rain(self, mask):
        m = mask & ~self.raining
        if m.any():
            self.raining[m] = True
            self.weather_timer[m] = self._weather_duration(m, np.ones(self.E, bool))[m]
        return m

    def wake_ground(self, mask, at=None):
        """A fissure: 'fissure_warn' ticks of warning (sensed on the vent channel), then open for a while."""
        c, f = self.c, self.f
        free = self.fz_state == 0
        mask = mask & free.any(1)
        if not mask.any():
            return np.zeros(self.E, bool)
        slot = np.argmax(free, 1)
        if at is None:
            if f["n_vents"] <= 0:
                return np.zeros(self.E, bool)
            k = np.minimum(np.floor(self._u(mask) * f["n_vents"]).astype(np.int64), f["n_vents"] - 1)
            ang = self._u(mask) * TWO_PI
            vr = self.vent_r[np.arange(self.E), k]
            dist = vr + c["fissure_vent_min"] + self._u(mask) * (c["fissure_vent_max"] - c["fissure_vent_min"])
            vp = self.vent_pos[np.arange(self.E), k]
            pts = vp + np.stack([dist * np.cos(ang), dist * np.sin(ang)], 1)
            fr = c["fissure_radius"]
            pts[:, 0] = np.clip(pts[:, 0], fr, c["width"] - fr); pts[:, 1] = np.clip(pts[:, 1], fr, c["height"] - fr)
            ok = mask.copy()
        else:
            pts = np.broadcast_to(np.asarray(at, np.float64), (self.E, 2)).copy()
            fr = c["fissure_radius"]
            ok = mask & (pts[:, 0] >= fr) & (pts[:, 1] >= fr) & (pts[:, 0] <= c["width"] - fr) & (pts[:, 1] <= c["height"] - fr)
        idx = np.nonzero(ok)[0]
        self.fz_pos[idx, slot[idx]] = pts[idx]; self.fz_state[idx, slot[idx]] = 1; self.fz_timer[idx, slot[idx]] = c["fissure_warn"]
        return ok

    def _perturb(self):
        f = self.f
        u = self._u(np.ones(self.E, bool))
        a1 = f["p_food"]; a2 = a1 + f["p_obs"]; a3 = a2 + f["p_vent"]; a4 = a3 + f["p_alert"]
        a5 = a4 + f["p_plant"]; a6 = a5 + f["p_raise"]; a7 = a6 + f["p_rain"]; a8 = a7 + f["p_fissure"]
        mf = u < a1
        mo = (u >= a1) & (u < a2)
        mv = (u >= a2) & (u < a3)
        ma = (u >= a3) & (u < a4)
        if mf.any():
            k = np.minimum(np.floor(self._u(mf) * f["food_n"]).astype(np.int64), f["food_n"] - 1)
            for i in range(f["food_n"]):
                mi = mf & (k == i) & self.food_on[:, i]
                if mi.any():
                    self._place_food(mi, i)
        if mo.any():
            self.insert_obstacle(mo)
        if mv.any() and f["n_vents"] > 0:
            k = np.minimum(np.floor(self._u(mv) * f["n_vents"]).astype(np.int64), f["n_vents"] - 1)
            for v in range(f["n_vents"]):
                self.wake_vent(mv & (k == v), v)
        if ma.any():
            self.alert_predator(ma)
        for lo, hi, act in ((a4, a5, self.plant_seed), (a5, a6, self.raise_thicket), (a6, a7, self.call_rain), (a7, a8, self.wake_ground)):
            mk = (u >= lo) & (u < hi)
            if mk.any():
                act(mk)

    # ----------------------------------------------------------------- predator (world rule, not learned)
    def _los_clear(self, a, b):
        """No slate intersects the segment a→b."""
        ab = b - a
        L2 = np.maximum(ab[:, 0] * ab[:, 0] + ab[:, 1] * ab[:, 1], 1e-12)
        ac = self.obs_pos - a[:, None, :]
        s = np.clip((ac[..., 0] * ab[:, None, 0] + ac[..., 1] * ab[:, None, 1]) / L2[:, None], 0.0, 1.0)
        cx = a[:, None, 0] + s * ab[:, None, 0] - self.obs_pos[..., 0]
        cy = a[:, None, 1] + s * ab[:, None, 1] - self.obs_pos[..., 1]
        return np.all(~self.obs_on | (cx * cx + cy * cy >= self.obs_r * self.obs_r), 1)

    @staticmethod
    def _push(p, center, minD, active):
        delta = p - center
        d = _norm(delta)
        hit = active & (d < minD)
        if not hit.any():
            return p
        safe = np.where(d > 1e-12, d, 1.0)
        nx = np.where(d > 1e-12, delta[:, 0] / safe, 1.0); ny = np.where(d > 1e-12, delta[:, 1] / safe, 0.0)
        return np.where(hit[:, None], center + np.stack([nx, ny], 1) * minD[:, None], p)

    def _predator_step(self):
        c, f = self.c, self.f
        on = self.pr_on
        if not on.any():
            return np.zeros(self.E, bool)
        light = self.light()
        detect = (c["detect_night"] + (c["detect_day"] - c["detect_night"]) * light) * f["detect_mult"]
        detect = np.where(self.raining, detect * c["rain_detect"], detect)
        vis = np.select([self.v < 0.02, self.v < 0.07, self.v < 0.15], [0.45, 0.65, 1.0], 1.35)
        d = _norm(self.pos - self.pr_pos)
        exposed = on & ~self.in_shelter & self._los_clear(self.pr_pos, self.pos)
        seen = exposed & (d < detect * vis)
        seen_ext = exposed & (d < detect * vis * c["lose_mult"])
        self.pr_seen = seen
        mode = self.pr_mode
        to_hunt = ((mode == PATROL) | (mode == SEARCH)) & seen
        tired = (mode == HUNT) & (self.pr_fatigue >= c["predator_fatigue_max"])
        lost = (mode == HUNT) & ~tired & ~seen_ext
        mode = np.where(to_hunt, HUNT, np.where(tired, REST, np.where(lost, SEARCH, mode)))
        self.pr_timer = np.where(tired, c["predator_rest_ticks"], np.where(lost, c["predator_search_ticks"], self.pr_timer))
        hunt = on & (mode == HUNT); search = on & (mode == SEARCH); patrol = on & (mode == PATROL); rest = on & (mode == REST)
        self.pr_last = np.where(hunt[:, None], self.pos, self.pr_last)
        target = np.where(hunt[:, None], self.pos, np.where(search[:, None], self.pr_last, self.pr_target))
        pounce = hunt & (d <= c["pounce_dist"])  # stalk slowly and quietly, then accelerate
        self.pr_pounce = pounce
        speed = np.select([pounce, hunt, search, patrol],
                          [self.pr_speed, c["predator_stalk"], c["predator_search"], c["predator_patrol"]], 0.0)
        speed = np.where(self._in_pool(self.pr_pos), speed * c["predator_water_factor"], speed)
        self.pr_fatigue = np.where(pounce, self.pr_fatigue + 1.0,
                                   np.where(hunt, self.pr_fatigue + 0.3, np.maximum(self.pr_fatigue - 0.5, 0.0)))
        self.pr_timer = np.where(search | rest, self.pr_timer - 1, self.pr_timer)
        delta = target - self.pr_pos
        dist = _norm(delta)
        step = np.minimum(speed, dist)
        mv = on & (dist > 1e-9) & (step > 0)
        safe = np.where(dist > 1e-9, dist, 1.0)
        self.pr_pos = np.where(mv[:, None], self.pr_pos + delta / safe[:, None] * step[:, None], self.pr_pos)
        self.pr_heading = np.where(mv, np.arctan2(delta[:, 1], delta[:, 0]), self.pr_heading)
        pr = c["predator_radius"]
        p = self.pr_pos
        for j in range(c["max_obstacles"]):
            p = self._push(p, self.obs_pos[:, j], self.obs_r[:, j] + pr, on & self.obs_on[:, j])
        for s in range(c["max_shelters"]):
            p = self._push(p, self.sh_pos[:, s], self.sh_r[:, s] + pr, on & self.sh_on[:, s] & (self.sh_h[:, s] > c["cover_min"]))
        p = np.stack([np.clip(p[:, 0], pr, c["width"] - pr), np.clip(p[:, 1], pr, c["height"] - pr)], 1)
        self.pr_pos = np.where(on[:, None], p, self.pr_pos)
        arrived_p = patrol & (_norm(self.pr_pos - self.pr_target) < 0.3)
        end_s = search & ((self.pr_timer <= 0) | (_norm(self.pr_pos - self.pr_last) < 0.3))
        end_r = rest & (self.pr_timer <= 0)
        mode = np.where(end_s | end_r, PATROL, mode)
        self.pr_fatigue = np.where(end_r, 0.0, self.pr_fatigue)
        self.pr_mode = mode
        need = arrived_p | end_s | end_r
        if need.any():
            self._new_waypoint(need)
        return on & ~self.in_shelter & (_norm(self.pos - self.pr_pos) < c["catch_dist"])

    # ----------------------------------------------------------------- step
    def step(self, action: np.ndarray):
        c = self.c
        a = np.asarray(action, np.int64)
        rc = c["creature_radius"]
        av = np.asarray(c["action_v"]); aw = np.asarray(c["action_w"])
        can_dash = self.stamina >= c["dash_min_stamina"]
        dashing = (a == A_DASH) & can_dash
        tv = np.where((a == A_DASH) & ~can_dash, av[0], av[a])
        tv = np.where(self.stamina < c["exhausted_stamina"], tv * c["exhausted_factor"], tv)
        tv = np.where(self.in_water, tv * c["water_factor"], tv)
        tw = aw[a]
        self.v = self.v + (tv - self.v) * c["accel_v"]
        self.w = self.w + (tw - self.w) * c["accel_w"]
        self.heading = np.mod(self.heading + self.w, TWO_PI)
        old = self.pos.copy()
        p = self.pos + self.v[:, None] * np.stack([np.cos(self.heading), np.sin(self.heading)], 1)
        contact = np.zeros(self.E, bool)
        for j in range(c["max_obstacles"]):
            q = self._push(p, self.obs_pos[:, j], rc + self.obs_r[:, j], self.obs_on[:, j])
            contact |= np.any(q != p, 1); p = q
        q = np.stack([np.clip(p[:, 0], rc, c["width"] - rc), np.clip(p[:, 1], rc, c["height"] - rc)], 1)
        contact |= np.any(q != p, 1)
        self.pos = q
        self.contact = contact
        self.ep["dist"] += _norm(q - old); self.ep["contact"] += contact
        self.in_shelter = self._cover(self.pos)
        self.in_water = self._in_pool(self.pos)
        self.resting = (a == A_REST) & (self.v < 0.02)
        self.energy -= np.where(self.resting, c["energy_decay_rest"], c["energy_decay"]) + c["move_cost_k"] * self.v * self.v
        moving = (a <= 2) | ((a == A_DASH) & ~can_dash)
        slow = (a >= 4) & (a <= 6)
        dst = np.select([dashing, moving, slow, a == A_REST],
                        [c["stamina_dash"], c["stamina_move"], c["stamina_slow"],
                         c["stamina_rest"] + c["stamina_rest_shelter"] * self.in_shelter], 0.0)
        dst = np.where((dst > 0) & (self.injury > 0), dst * c["injury_recovery"], dst)  # injured bodies recover slowly
        self.stamina = np.clip(self.stamina + dst, 0.0, 1.0)
        reward = np.zeros(self.E)
        dv = _norm(self.vent_pos - self.pos[:, None, :])
        dfz = _norm(self.fz_pos - self.pos[:, None, :])
        inh = np.any(self.vent_on & (dv < self.vent_r), 1) | np.any((self.fz_state == 2) & (dfz < c["fissure_radius"]), 1)
        self.injury = np.where(inh, c["injury_ticks"], np.maximum(self.injury - 1, 0))
        self.ep["injury"] += inh
        self.in_hazard = inh
        self.energy -= c["hazard_drain"] * inh
        reward += c["reward_hazard_tick"] * inh
        self.ep["hazard"] += inh
        for i in range(c["max_food"]):
            d = _norm(self.food_pos[:, i] - self.pos)
            eat = self.food_on[:, i] & (d < rc + c["food_radius"])
            if eat.any():
                self.energy = np.where(eat, np.minimum(1.0, self.energy + c["food_gain"]), self.energy)
                reward += c["reward_food"] * eat
                self.ep["food"] += eat
                for e in np.nonzero(eat)[0][:64]:
                    self.events.append((int(e), int(self.t[e]), "eat", i))
                self.food_on[eat, i] = False
                self.food_timer[eat, i] = (c["regrow_min"] + np.floor(self._u(eat) * c["regrow_rand"]))[eat]
        for i in range(self.f["food_n"]):  # seeds regrow where the meadow allows, after a delay
            off = ~self.food_on[:, i]
            if not off.any():
                continue
            k = i % self.c["n_meadows"]  # regrowth follows the meadow's fertility cycle and rain; the hollow is richer
            rate = self.fertility(k) * (self.c["hollow_rate"] if (k == 0 and self.f["n_vents"] > 0) else 1.0)
            self.food_timer[off, i] -= rate[off]
            grow = off & (self.food_timer[:, i] <= 0)
            if grow.any():
                self._place_food(grow, i)
                self.food_on[grow, i] = True
                for e in np.nonzero(grow)[0][:64]:
                    self.events.append((int(e), int(self.t[e]), "grow", i))
        for v in range(c["max_vents"]):
            pv = self.vent_present[:, v]
            if not pv.any():
                continue
            self.vent_timer[pv, v] -= 1
            flip = pv & (self.vent_timer[:, v] <= 0)
            if flip.any():
                self.vent_on[flip, v] = ~self.vent_on[flip, v]
                self.vent_timer[flip, v] = self._vent_duration(flip, self.vent_on[:, v])[flip]
        g = c["max_food"] - 1  # planted seed ripens
        rip = self.plant_timer > 0
        if rip.any():
            self.plant_timer[rip] -= 1
            ready = rip & (self.plant_timer <= 0)
            self.food_on[ready, g] = True
        self.weather_timer -= 1
        wflip = self.weather_timer <= 0
        if wflip.any():
            self.raining[wflip] = ~self.raining[wflip]
            self.weather_timer[wflip] = self._weather_duration(wflip, self.raining)[wflip]
        # thickets: dry spells wilt, rain restores; a raised thicket grows, lives, then withers away
        perm = self.sh_on & (self.sh_life < 0)
        self.sh_h = np.where(perm, np.clip(self.sh_h + np.where(self.raining, c["thicket_rain_rate"], -c["thicket_dry_rate"])[:, None], 0.0, 1.0), self.sh_h)
        growing = self.sh_on & (self.sh_life > 0)
        self.sh_h = np.where(growing, np.minimum(1.0, self.sh_h + 1.0 / c["raised_grow"]), self.sh_h)
        self.sh_life = np.where(growing, self.sh_life - 1, self.sh_life)
        dying = self.sh_on & (self.sh_life == 0)
        self.sh_h = np.where(dying, self.sh_h - 1.0 / c["raised_wither"], self.sh_h)
        gone = dying & (self.sh_h <= 0)
        self.sh_on = self.sh_on & ~gone; self.sh_h = np.where(gone, 0.0, self.sh_h); self.sh_life = np.where(gone, -1, self.sh_life)
        # fissures: warning → open → closed
        act = self.fz_state > 0
        self.fz_timer = np.where(act, self.fz_timer - 1, self.fz_timer)
        opening = (self.fz_state == 1) & (self.fz_timer <= 0)
        closing = (self.fz_state == 2) & (self.fz_timer <= 0)
        if opening.any():
            for j in range(c["max_fissures"]):
                oj = opening[:, j]
                if oj.any():
                    self.fz_timer[oj, j] = (c["fissure_open_min"] + np.floor(self._u(oj) * c["fissure_open_rand"])).astype(np.int64)[oj]
        self.fz_state = np.where(opening, 2, np.where(closing, 0, self.fz_state))
        self.in_shelter = self._cover(self.pos)
        caught = self._predator_step()
        threat = self.pr_on & ((self.pr_mode == HUNT) | (self.pr_mode == SEARCH)) & (_norm(self.pos - self.pr_pos) < 6.0)
        night = self.light() < 0.3
        self.ep["threat"] += threat; self.ep["shelter_threat"] += threat & self.in_shelter
        self.ep["shelter"] += self.in_shelter; self.ep["rest"] += self.resting
        self.ep["dash"] += dashing; self.ep["dash_threat"] += dashing & threat
        self.ep["night"] += night; self.ep["shelter_night"] += night & self.in_shelter
        self._perturb()
        self.t += 1
        self.last_action = a
        starve = self.energy <= 0
        self.energy = np.maximum(self.energy, 0.0)
        dead = starve | caught
        reward += np.where(caught, c["reward_caught"], np.where(starve, c["reward_starve"], 0.0))
        self.ep["caught"] += caught
        trunc = (~dead) & (self.t >= c["max_ticks"])
        self.ep["ret"] += reward
        self._update_memory(np.ones(self.E, bool))
        return reward, dead, trunc

    # ----------------------------------------------------------------- observation
    def _predator_percept(self):
        """What Vela senses of the predator right now: seen on a ray (light-limited sight) or heard. Shared by memory and observe."""
        c = self.c
        light = self.light()
        ang = self.heading[:, None] + self.ray_rel[None, :]
        dx, dy = np.cos(ang), np.sin(ang)
        ox = (self.pr_pos[:, 0] - self.pos[:, 0])[:, None]; oy = (self.pr_pos[:, 1] - self.pos[:, 1])[:, None]
        proj = ox * dx + oy * dy; perp2 = ox * ox + oy * oy - proj * proj; r2 = c["predator_radius"] ** 2
        thc = np.sqrt(np.maximum(r2 - perp2, 0.0)); t0 = proj - thc; t1 = proj + thc
        t = np.where(perp2 <= r2, np.where(t0 >= 0, t0, np.where(t1 >= 0, 0.0, np.inf)), np.inf)
        sight = c["pred_sight_night"] + (c["pred_sight_day"] - c["pred_sight_night"]) * light
        seen = self.pr_on & np.any(t <= sight[:, None], 1)
        hr = np.where(self.pr_mode == HUNT, np.where(self.pr_pounce, c["hear_hunt"], c["hear_stalk"]), c["hear_calm"])
        hr = np.where(self.raining, hr * c["rain_hear"], hr)
        dpr = _norm(self.pr_pos - self.pos)
        heard = self.pr_on & (self.pr_mode != REST) & (dpr < hr)
        return seen, heard

    def _update_memory(self, mask):
        seen, heard = self._predator_percept()
        sensed = mask & (seen | heard)
        self.mem_t = np.where(mask, np.where(sensed, 0, self.mem_t + 1), self.mem_t)
        self.mem_pos = np.where(sensed[:, None], self.pr_pos, self.mem_pos)
        self.mem_pounce = np.where(sensed, self.pr_pounce & (self.pr_mode == HUNT), self.mem_pounce)

    def memory_features(self):
        """5 inputs: recency exp(-Δt/τ); sin, cos of the bearing to where it was last sensed (relative to current heading);
        remembered proximity; whether it was pouncing. All scaled by recency (0 when never sensed)."""
        c = self.c
        rec = np.exp(-np.minimum(self.mem_t, 10**5) / c["memory_tau"])
        rel = self.mem_pos - self.pos
        b = np.arctan2(rel[:, 1], rel[:, 0]) - self.heading
        prox = np.maximum(0.0, 1.0 - _norm(rel) / c["memory_range"])
        return np.stack([rec, np.sin(b) * rec, np.cos(b) * rec, prox * rec, self.mem_pounce * rec], 1)

    def observe(self) -> np.ndarray:
        c = self.c
        R, rng_ = c["n_rays"], c["ray_range"]
        light = self.light()
        ang = self.heading[:, None] + self.ray_rel[None, :]
        dx, dy = np.cos(ang), np.sin(ang)
        px, py = self.pos[:, 0:1], self.pos[:, 1:2]
        with np.errstate(divide="ignore", invalid="ignore"):
            tx = np.where(dx > 1e-12, (c["width"] - px) / dx, np.where(dx < -1e-12, (0.0 - px) / dx, np.inf))
            ty = np.where(dy > 1e-12, (c["height"] - py) / dy, np.where(dy < -1e-12, (0.0 - py) / dy, np.inf))
        t_wall = np.minimum(tx, ty)

        def circles(cpos, cr, con):
            ox = cpos[:, None, :, 0] - px[:, :, None]
            oy = cpos[:, None, :, 1] - py[:, :, None]
            proj = ox * dx[:, :, None] + oy * dy[:, :, None]
            d2 = ox * ox + oy * oy
            perp2 = d2 - proj * proj
            r2 = (cr * cr)[:, None, :]
            hit = con[:, None, :] & (perp2 <= r2)
            thc = np.sqrt(np.maximum(r2 - perp2, 0.0))
            t0 = proj - thc; t1 = proj + thc
            t = np.where(t0 >= 0, t0, np.where(t1 >= 0, 0.0, np.inf))
            return np.where(hit, t, np.inf).min(axis=2)

        t_solid = np.minimum(t_wall, circles(self.obs_pos, self.obs_r, self.obs_on))
        t_food = circles(self.food_pos, np.full(self.food_pos.shape[:2], c["food_radius"]), self.food_on)
        prox = lambda t: 1.0 - np.minimum(t, rng_) / rng_

        def circles_val(cpos, cr, con, val):  # strongest (proximity × value) over items on each ray
            ox = cpos[:, None, :, 0] - px[:, :, None]
            oy = cpos[:, None, :, 1] - py[:, :, None]
            proj = ox * dx[:, :, None] + oy * dy[:, :, None]
            perp2 = ox * ox + oy * oy - proj * proj
            r2 = (cr * cr)[:, None, :]
            hit = con[:, None, :] & (perp2 <= r2)
            thc = np.sqrt(np.maximum(r2 - perp2, 0.0))
            t0 = proj - thc; t1 = proj + thc
            t = np.where(hit, np.where(t0 >= 0, t0, np.where(t1 >= 0, 0.0, np.inf)), np.inf)
            return (prox(t) * val[:, None, :]).max(axis=2)

        fz_val = np.where(self.fz_state == 2, 1.0, np.where(self.fz_state == 1, c["fissure_warn_signal"], 0.0))
        v_haz = np.maximum(prox(circles(self.vent_pos, self.vent_r, self.vent_on)),
                           circles_val(self.fz_pos, np.full(self.fz_state.shape, c["fissure_radius"]), self.fz_state > 0, fz_val))
        v_sh = circles_val(self.sh_pos, self.sh_r, self.sh_on, self.sh_h)
        t_wat = circles(self.pool_pos, self.pool_r, self.pool_on)
        t_pr = circles(self.pr_pos[:, None, :], np.full((self.E, 1), c["predator_radius"]), self.pr_on[:, None])
        sight = c["pred_sight_night"] + (c["pred_sight_day"] - c["pred_sight_night"]) * light
        t_pr = np.where(t_pr <= sight[:, None], t_pr, np.inf)
        rays = np.stack([prox(t_solid), prox(t_food), v_haz, v_sh, prox(t_pr), prox(t_wat)], 2).reshape(self.E, R * N_CH)
        hr = np.where(self.pr_mode == HUNT, np.where(self.pr_pounce, c["hear_hunt"], c["hear_stalk"]), c["hear_calm"])
        hr = np.where(self.raining, hr * c["rain_hear"], hr)
        rel = self.pr_pos - self.pos
        dpr = _norm(rel)
        heard = self.pr_on & (self.pr_mode != REST) & (dpr < hr)
        hp = np.where(heard, 1.0 - dpr / hr, 0.0)
        bearing = np.arctan2(rel[:, 1], rel[:, 0]) - self.heading
        phase = TWO_PI * (self.t + self.t0) / c["day_ticks"]
        onehot = np.zeros((self.E, N_ACTIONS)); onehot[np.arange(self.E), self.last_action] = 1.0
        selfv = np.stack([self.energy, self.stamina, self.v / c["v_max"], self.w / c["w_max"], self.in_shelter * 1.0,
                          self.in_hazard * 1.0, self.in_water * 1.0, self.contact * 1.0, light, self.raining * 1.0,
                          np.sin(phase), np.cos(phase),
                          hp, np.where(heard, np.sin(bearing), 0.0), np.where(heard, np.cos(bearing), 0.0),
                          self.injury / c["injury_ticks"]], 1)
        return np.concatenate([rays, selfv, self.memory_features(), onehot], 1)


class VecEnv:
    """Auto-resetting batch env. Episode seeds come from a numpy Generator (non-world randomness)."""

    def __init__(self, n_envs, family="train", seed=0, seed_lo=0, seed_hi=None):
        self.w = World(n_envs, family)
        self.n = n_envs
        self.seed_rng = np.random.default_rng(seed)
        self.seed_lo = seed_lo
        self.seed_hi = seed_hi if seed_hi is not None else C["train_seed_max"]
        self.w.reset(np.ones(n_envs, bool), self._seeds())
        self.w.events.clear()

    def _seeds(self):
        return self.seed_rng.integers(self.seed_lo, self.seed_hi, self.n, dtype=np.int64)

    def obs(self):
        return self.w.observe()

    def step(self, action):
        r, dead, trunc = self.w.step(action)
        done = dead | trunc
        info = None
        if done.any():
            w = self.w
            info = {"food": w.ep["food"][done].copy(), "ticks": w.t[done].copy(), "ret": w.ep["ret"][done].copy(),
                    "dead": dead[done].copy(), "caught": w.ep["caught"][done].copy()}
            w.reset(done, self._seeds())
        self.w.events.clear()
        return self.obs(), r, dead, trunc, info
